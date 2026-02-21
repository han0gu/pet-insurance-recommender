from langchain_core.documents import Document

chunk = Document(
    page_content=("지급된 것으로 보고 최종 후유장해 상태에 해당되는 후유장해</p><br><h1 id='52' "
 "style='font-size:14px'>보험금에서 이를 차감하여 지급합니다.</h1><br><table id='53' "
 "style='font-size:14px'><thead><tr><td>예 시</td><td>장해지급률 "
 '계산</td></tr></thead><tbody><tr><td colspan="2">① 보험가입 전 한 다리의 관절에 약간의 장해(지급률 '
 '5%)가 있었던 피보험자 가 보험가입 후 상해로 그 다리의 해당관절이 기능을 완전히'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000392',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
