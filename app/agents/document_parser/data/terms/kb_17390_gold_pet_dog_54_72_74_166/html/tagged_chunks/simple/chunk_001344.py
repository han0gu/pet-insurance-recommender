from langchain_core.documents import Document

chunk = Document(
    page_content=("청구를 받더라도 회사는 이를 지급하<br>지 않습니다.</p><br><p id='214' data-category='paragraph' "
 "style='font-size:16px'>제5조(지정대리청구인의 변경지정)</p><br><p id='215' "
 "data-category='paragraph' style='font-size:16px'>계약자는 다음의 서류를</p><br><p "
 "id='216' data-category='paragraph' style='font-size:16px'>제출하고 지정대리청구인을 변경 "
 '지정할 수 있습니다'),
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
 'indexing': {'chunk_id': 'chunk_001344',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
