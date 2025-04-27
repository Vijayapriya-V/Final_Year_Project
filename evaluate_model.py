import pandas as pd
import matplotlib.pyplot as plt

def genre_wise_captioning_accuracy():
    # Example genre names and their corresponding accuracy scores (e.g., BLEU, METEOR, CIDEr)
    genres = ['Indoor', 'Nature', 'Social', 'Sports', 'Urban']  # Adjust based on your dataset

    # Image captioning scores for each genre (replace with actual results)
    image_bleu_scores = [0.75, 0.82, 0.69, 0.88, 0.74]
    image_meteor_scores = [0.68, 0.74, 0.65, 0.79, 0.72]
    image_cider_scores = [1.2, 1.4, 1.0, 1.6, 1.3]

    # Video captioning scores for each genre (replace with actual results)
    video_bleu_scores = [0.78, 0.84, 0.72, 0.90, 0.76]
    video_meteor_scores = [0.71, 0.77, 0.70, 0.82, 0.75]
    video_cider_scores = [1.3, 1.5, 1.1, 1.8, 1.4]

    # Create a DataFrame to display the scores as a table
    data = {
        'Genre': genres,
        'Image - BLEU': image_bleu_scores,
        'Video - BLEU': video_bleu_scores,
        'Image - METEOR': image_meteor_scores,
        'Video - METEOR': video_meteor_scores,
        'Image - CIDEr': image_cider_scores,
        'Video - CIDEr': video_cider_scores
    }
    
    df = pd.DataFrame(data)

    # Print the table
    print(df)

    # Save the table as a CSV file
    df.to_csv('captioning_comparison_table.csv', index=False)

    # Plot the table as an image (optional)
    fig, ax = plt.subplots(figsize=(10, 4))  # Set table size
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    # Save the table as an image
    plt.savefig('captioning_comparison_table.png', bbox_inches='tight', dpi=300)
    plt.show()

# Call the function to generate and save the table
if __name__ == "__main__":
    genre_wise_captioning_accuracy()
