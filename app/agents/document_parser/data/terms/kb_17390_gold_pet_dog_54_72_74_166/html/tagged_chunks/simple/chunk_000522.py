from langchain_core.documents import Document

chunk = Document(
    page_content=("의료법 제3조(의료기관) 제2항에서 정한 국내의 병</p><br><p id='260' data-category='paragraph' "
 "style='font-size:16px'>원이나 의원 또는 국외의 의료관련법에서 정한 의료기관을 말합니다.</p><br><table "
 "id='261' style='font-size:16px'><thead></thead><tbody><tr><td>용 어 "
 '풀</td><td>이</td></tr><tr><td colspan="2">∙ 절단 : 특정부위를 잘라 내는 것 ∙ 절제 : 특정부위를 '
 '잘라 없애는 것 ∙'),
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
 'indexing': {'chunk_id': 'chunk_000522',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
