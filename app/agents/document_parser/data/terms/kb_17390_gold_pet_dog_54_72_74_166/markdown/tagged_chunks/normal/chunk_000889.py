from langchain_core.documents import Document

chunk = Document(
    page_content=('- 4) 갈비뼈(늑골)의 기형은 그 개수와 정도, 부위 등에 관계없이 전체를 일괄\n'
 '- 하여 하나의 장해로 취급한다. 다발성늑골 기형의 경우 각각의 각(角) 변\n'
 '- 형을 합산하지 않고 그 중 가장 높은 각(角) 변형을 기준으로 평가한다.\n'
 '- 146 -- \n'
 '![image](/image/placeholder)\n'
 '부 가 설 명 가슴뼈부 가 설 명골반뼈![image](/image/placeholder)\n'
 '# 팔의 장해8.| 가. 장해의 분류 | 공 통 |\n'
 '| --- | --- |\n'
 '| 장해의 분류 | 지급률 사항 |'),
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
 'indexing': {'chunk_id': 'chunk_000889',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
