from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제1항에서 지정한 특정질병의 합병증으로 인해 발생한 특정질병이외의 질병<br>으로 보험계약에서 정한 보험금의 지급사유가 발생한 '
 '경우<br>2. 상해를 직접적인 원인으로 하여 보험금의 지급사유가 발생한 경우<br>3'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001454',
              'chunk_char_len': 121,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
