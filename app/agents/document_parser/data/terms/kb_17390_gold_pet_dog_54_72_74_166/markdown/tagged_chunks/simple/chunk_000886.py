from langchain_core.documents import Document

chunk = Document(
    page_content=('- 14) ‘추간판탈출증으로 인한 약간의 신경 장해’란 추간판탈출증이 확인되\n'
 '- 고 신경생리검사에서 명확한 신경근병증의 소견이 지속되는 경우\n'
 '146 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)- \n'
 '- \n'
 '7.# 체간골의 장해| 가. 장해의 분류 |  |\n'
 '| --- | --- |\n'
 '| 장해의 분류 1) 어깨뼈(견갑골)나 골반뼈(장골, 제2천추 이하의 천골, 미골, 좌골 포함)에 뚜렷한 기형을 남긴 때 | 지급률 15 '
 '|\n'
 '| 2) 빗장뼈(쇄골), 가슴뼈(흉골), 갈비뼈(늑골)에 뚜렷한 기형을 | 10 |\n'
 '# 남긴 때-'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000886',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
