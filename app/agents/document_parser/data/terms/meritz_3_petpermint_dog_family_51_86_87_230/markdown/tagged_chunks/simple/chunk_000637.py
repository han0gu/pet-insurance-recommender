from langchain_core.documents import Document

chunk = Document(
    page_content=('- 탈출증이 확인되고 신경생리검사에서 명확한 신경근병\n'
 '- 증의 소견이 지속되는 경우\n'
 '# 7. 체간골의 장해# 가. 장해의 분류| 장해의 분류 | 지급률 |\n'
 '| --- | --- |\n'
 '| 1) 어깨뼈(견갑골)나 골반뼈(장골, 제2천추 이하 의 천골, 미골, 좌골 포함)에 뚜렷한 기형을 남긴 때 2) 빗장뼈(쇄골), '
 '가슴뼈(흉골), 갈비뼈(늑골)에 뚜렷한 기형을 남긴 때 | 15 10 |\n'
 '# 나. 장해판정기준- 1) “체간골”이라 함은 어깨뼈(견갑골), 골반뼈(장골,\n'
 '- 제2천추 이하의 천골, 미골, 좌골 포함), 빗장뼈(쇄'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000637',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
