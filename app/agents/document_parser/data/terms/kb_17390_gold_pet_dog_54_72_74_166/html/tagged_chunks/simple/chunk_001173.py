from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 이 특별약관이 의무보<br>험인 경우에는 그러하지 않으며, 의무보험이 다수인 경우에는 제11조(보험금의<br>분담)을 '
 '따릅니다.<br>\uf000 제1항의 의무보험은 피보험자가 법률에 따라 의무적으로 가입하여야 하는 보험으<br>로써 공제계약을 '
 '포함합니다.<br>\uf000 피보험자가 의무보험에 가입하여야 함에도 불구하고 가입하지 않은 경우에는 그<br>가 가입했더라면 '
 '의무보험에서 보상했을 금액을 제1항의 "의무보험에서 보상하는</p><br><h1 id=\'187\' '
 'style=\'font-size:14px\'>금액"으로'),
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
 'indexing': {'chunk_id': 'chunk_001173',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
