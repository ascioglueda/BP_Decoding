    def finish(self):
        #if self.id == REQUESTING_NODE:
        if self.decoded:
                self.log(f'Simülasyon basariyla tamamlandi. Basari orani: {np.mean(self.known_values == original_data):.2f}')
        else:
                self.log(f'Simülasyon tamamlandi, ancak tum veri cozulemedi. Cozulemeyen semboller: {np.sum(self.known_values == 1)}')