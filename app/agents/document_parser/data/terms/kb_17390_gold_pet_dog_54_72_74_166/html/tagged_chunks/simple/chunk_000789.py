from langchain_core.documents import Document

chunk = Document(
    page_content=('. 피보험자 본인 또는 배우자와 생계를 같이하는 별거 중인 미혼자녀<br>관 련 법 규 민법<br>∙ 제777조(친족의 범위)에서 규정한 '
 "친족의 범위</p><br><h1 id='155' style='font-size:14px'>8촌 이내의 혈족, 4촌 이내의 인척, "
 "배우자</h1><h1 id='156' style='font-size:14px'>제4조(보험금의 청구)</h1><br><p id='157' "
 "data-category='list' style='font-size:14px'>\uf000 보험수익자는 다음의 서류를 제출하고 보험금을"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000789',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
