from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '|  | 회사가 영업점에서 정상적으로 영업하는 날을 말하며, 토 |\n'
 '| 영업일 | 요일, "관공서의 공휴일에 관한 규정"에 따른 공휴일과 노 동절을 제외합니다. |\n'
 '|  | (대통령령 제31930호) |\n'
 '| --- | --- |\n'
 '# 관 련 법 규# 관공서의 공휴일에 관한 규정 제2조 및 제3조별\n'
 '제2조(공휴일)\n'
 '약\n'
 '관공서의 공휴일은 다음 각 호와 같다. 다만, 재외공관의 공휴일은 우리나라의- 국경일 중 공휴일과 주재국의 공휴일로 한다.\n'
 '- 1. 일요일'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000452',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
