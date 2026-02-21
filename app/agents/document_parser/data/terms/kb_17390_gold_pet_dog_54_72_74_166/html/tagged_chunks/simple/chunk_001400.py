from langchain_core.documents import Document

chunk = Document(
    page_content=("특례) 제1항에 정한 방법으로 보험계약 안내자료를 수</p><br><p id='54' data-category='paragraph' "
 "style='font-size:14px'>령하고자 하는 경우 계약을 청약할 때 보험계약 안내자료를 수령할 전자우편(이메<br>일) 및 "
 '전자적 의사표시로 제공될 주소(이하 "전자적 주소"라 합니다)를 지정하여</p><br><p id=\'55\' '
 "data-category='paragraph' style='font-size:14px'>136 KB 금쪽같은 "
 '펫보험(강아지)(무배당)(26.01)</p><p'),
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
 'indexing': {'chunk_id': 'chunk_001400',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
