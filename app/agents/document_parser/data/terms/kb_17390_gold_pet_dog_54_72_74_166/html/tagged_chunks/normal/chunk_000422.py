from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>수술 1cm당 7만원<br>상해흉터복원수술비 수술 1cm당 14만원<br>(단, 3cm이상의 "
 "경우에 한함)<br>주) 길이측정이 불가한 피부이식수술 등의 경우 수술cm는 최장직경으로 함 제</p><br><p id='99' "
 "data-category='paragraph' style='font-size:14px'>\uf000 제1항에서 정한 안면부, 상지, "
 '하지란 다음을 말합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'skin']},
 'indexing': {'chunk_id': 'chunk_000422',
              'chunk_char_len': 228,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
