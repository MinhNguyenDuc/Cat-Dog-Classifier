from django import forms


class ImageForm(forms.Form):
    image_url = forms.CharField(max_length=500,widget=forms.TextInput(attrs={
        'class': 'form-control',
        'placeholder': 'Enter image url'
    }))
