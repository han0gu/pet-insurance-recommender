from langchain_core.documents import Document

chunk = Document(
    page_content=('암의 치료를 직접<br>적인 목적으로 화학요법 항암제 또는 Tyrosine kinase inhibitor(TKI) 표적항암<br>제를 '
 '사용하여 시행한 치료(정맥, 피하 또는 경구 등을 통해서 투여하는 약제를<br>사용한 치료를 포함합니다.)를 말합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001035',
              'chunk_char_len': 140,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
