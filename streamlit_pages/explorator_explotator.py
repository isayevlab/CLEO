import os
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from umap import UMAP
from modules.preprocess_data import preprocess_data

def explorator_explotator():
    st.title("Exploration and Exploitation")

    experiment_name = st.text_input(label="Experiment ID", placeholder="polymer_space_1")
    sample_number = st.number_input(label="Desired number of samples", value=20)
    n_clusters = st.slider("Number of clusters for diverse selection", min_value=1, max_value=max(2, int(sample_number)), step=1)

    if experiment_name:
        space_path = f"experiments/{experiment_name}/space.csv"
        predictions_path = f"experiments/{experiment_name}/predictions.csv"

        # Check if the space exists
        if not os.path.exists(space_path):
            st.error(f"The space file does not exist for the experiment '{experiment_name}' in '{space_path}'. Please verify the experiment ID.")
            return  # Stop further execution

        if not os.path.exists(predictions_path):
            st.error(f"The predictions file does not exist for the experiment '{experiment_name}' in '{predictions_path}'. Please verify the experiment ID.")
            return  # Stop further execution

        space = pd.read_csv(space_path, index_col=0)
        predictions = pd.read_csv(predictions_path, index_col=0)

        # Allow user to fix variables
        with st.expander("Fix Variables"):
            fixed_columns = st.multiselect("Select columns to fix", options=space.columns if 'space' in locals() else [])
            fixed_values = {}
            for col in fixed_columns:
                unique_values = space[col].unique()
                fixed_values[col] = st.selectbox(f"Select value for {col}", options=unique_values)

            if fixed_columns:
                query = " & ".join([f"{col} == @fixed_values['{col}']" for col in fixed_columns])
                space = space.query(query)

        ignore_columns = [col for col in space.columns if col.startswith("__")]
        space_filtered = space.loc[:, ~space.columns.isin(ignore_columns)].copy()
        features = list(space_filtered.columns)

        space_filtered["Prediction"] = predictions["Prediction"]
        space_filtered["Uncertainty"] = predictions["Uncertainty"]

        # Regression Task
        if len(pd.unique(space_filtered["Prediction"])) > 10:
            task_type = "regression"
            fraction = st.slider("Fraction of space used for search", min_value=0.01, max_value=1.0, step=0.01)
            threshold = predictions["Prediction"].quantile(1 - fraction)
            reduced_space = space_filtered[space_filtered["Prediction"] >= threshold]

            # Preprocess data and cluster the reduced space
            processed_space = preprocess_data(reduced_space[features])
            clusters = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(processed_space)
            reduced_space["clusters"] = clusters

            # Calculate points per cluster
            points_per_cluster = [sample_number // n_clusters + (i < sample_number % n_clusters) for i in range(n_clusters)]
            assert sum(points_per_cluster) == sample_number

            # Select samples based on clustering
            selected = []
            for cluster in range(n_clusters):
                cluster_space = reduced_space[reduced_space["clusters"] == cluster]
                selected.append(cluster_space.nlargest(points_per_cluster[cluster], "Prediction"))

            selected = pd.concat(selected)
            selected.sort_values(by="Prediction", inplace=True, ascending=False)

        # Classification Task
        else:
            task_type = "classification"
            # Assume only one class is selected
            selected_class = st.selectbox("Select Class for Sampling", space_filtered["Prediction"].unique())
            sort_by_uncertainty = st.radio("Sort by Uncertainty",
                                           options=["Higher Uncertainty", "Lower Uncertainty"])
            ascending_order = sort_by_uncertainty == "Lower Uncertainty"

            # Filter space for the selected class
            reduced_space = space_filtered[space_filtered["Prediction"] == selected_class]
            reduced_space = reduced_space.sort_values(by="Uncertainty", ascending=ascending_order)

            # Preprocess data and cluster the reduced space
            processed_space = preprocess_data(reduced_space[features])
            clusters = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(processed_space)
            reduced_space["clusters"] = clusters

            # Calculate points per cluster
            points_per_cluster = [sample_number // n_clusters + (i < sample_number % n_clusters) for i in range(n_clusters)]
            assert sum(points_per_cluster) == sample_number

            # Select samples based on clustering
            selected = []
            for cluster in range(n_clusters):
                cluster_space = reduced_space[reduced_space["clusters"] == cluster]
                selected.append(cluster_space.nsmallest(points_per_cluster[cluster], "Uncertainty") if ascending_order else cluster_space.nlargest(points_per_cluster[cluster], "Uncertainty"))

            selected = pd.concat(selected)
            selected.sort_values(by="Prediction", inplace=True, ascending=False)

        # Assign colors for exploration and selection in the full space
        space_filtered["Color"] = "Did not explore"
        space_filtered.loc[reduced_space.index, "Color"] = "Explored"
        space_filtered.loc[selected.index, "Color"] = "Selected"

        # Scatter plot of Uncertainty vs Prediction
        fig = px.scatter(space_filtered,
                         x="Uncertainty",
                         y="Prediction",
                         color="Color",
                         color_discrete_map={"Did not explore": 'grey',
                                             "Explored": 'blue',
                                             "Selected": 'red'},
                         hover_data=features)
        st.plotly_chart(fig)

        # UMAP Projection for Clustering Visualization
        projected_data = UMAP(n_neighbors=5, min_dist=0.5, metric='cosine').fit_transform(processed_space.values)
        reduced_space["UMAP 1"] = projected_data[:, 0]
        reduced_space["UMAP 2"] = projected_data[:, 1]

        reduced_space["Color"] = clusters
        reduced_space["Color"] = "Cluster " + reduced_space["Color"].astype(str)
        reduced_space.loc[selected.index, "Color"] = "Selected"

        # UMAP plot for clusters
        fig = px.scatter(reduced_space,
                         x="UMAP 1",
                         y="UMAP 2",
                         color="Color",
                         hover_data=features)
        st.plotly_chart(fig)

        # Include ignored columns in the final output
        ignored_columns_data = space.loc[selected.index, ignore_columns]
        final_output = pd.concat([selected, ignored_columns_data], axis=1)

        # Display selected samples
        st.dataframe(final_output)

if __name__ == "__main__":
    explorator_explotator()
