from langchain_core.documents import Document

chunk = Document(
    page_content=('않습니다.<br>\uf000 제1항에서 보험증권을 받은 날에 대한 다툼이 발생한 경우 회사가 이를 증명하여야</p><br><p '
 "id='202' data-category='paragraph' style='font-size:16px'>합니다.</p><br><h1 "
 "id='203' style='font-size:16px'>제20조(약관교부 및 설명의무</h1><br><p id='204' "
 "data-category='paragraph' style='font-size:16px'>등)</p><br><p id='205'"),
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
 'indexing': {'chunk_id': 'chunk_000164',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
