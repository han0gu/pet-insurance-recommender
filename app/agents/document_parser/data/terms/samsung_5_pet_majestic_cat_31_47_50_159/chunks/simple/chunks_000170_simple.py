from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[현저하게 공정을 잃은 합의]\n'
 '회사가 보험수익자의 경제적․신체적․정신적인 어려움, 경솔함, 경험 부족 등을 이용하여 동일․유사 사례에 비추어 보험수익자에게 매우 '
 '불합리하게 합의를 하는 것을 의미합니다.\n'
 '제45조 (개인정보보호)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 46},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000170',
              'chunk_char_len': 133,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
