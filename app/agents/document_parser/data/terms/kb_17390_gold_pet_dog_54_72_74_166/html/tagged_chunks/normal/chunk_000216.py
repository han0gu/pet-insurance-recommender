from langchain_core.documents import Document

chunk = Document(
    page_content=("어 풀 이</p><br><p id='25' data-category='paragraph' "
 "style='font-size:16px'>미경과보험료</p><br><table id='26' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>미경과보험료는 "
 '보험계약자가</td><td>납입한 보험료 중에서 아직 당해 보험료 기간이 경</td><td>ㆍ 규정</td></tr><tr><td '
 'colspan="3">과되지 않은 보험료를 말합니다'),
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
 'indexing': {'chunk_id': 'chunk_000216',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
