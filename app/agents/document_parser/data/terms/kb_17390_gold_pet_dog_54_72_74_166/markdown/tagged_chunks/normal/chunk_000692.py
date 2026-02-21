from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- | --- | --- |\n'
 '| 을 공제한 액수를 말합니다. 제 | 을 공제한 액수를 말합니다. 제 | 을 공제한 액수를 말합니다. 제 | 을 공제한 액수를 '
 '말합니다. 제 |\n'
 '해# 제15조(대위권)# \uf000 회사가 보험금을지급한 때(현물보상한 경우를 포함합니다)에는 회사는 지급한및보험금의 한도내에서 아래의 '
 '권리를 가집니다. 다만, 회사가 보상한 금액이 피보 특'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000692',
              'chunk_char_len': 210,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
