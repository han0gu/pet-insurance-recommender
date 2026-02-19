from langchain_core.documents import Document

chunk = Document(
    page_content=('7. 체간골의 장해\n'
 '가. 장해의 분류\n'
 '장해의 분류 | 지급률\n'
 '1) 어깨뼈(견갑골)나 골반뼈(장골, 제2천추 이하 의 천골, 미골, 좌골 포함)에 뚜렷한 기형을 남긴 때 2) 빗장뼈(쇄골), '
 '가슴뼈(흉골), 갈비뼈(늑골)에 뚜렷한 기형을 남긴 때 | 15 10\n'
 '나. 장해판정기준\n'
 '1) “체간골”이라 함은 어깨뼈(견갑골), 골반뼈(장골, 제2천추 이하의 천골, 미골, 좌골 포함), 빗장뼈(쇄 골), 가슴뼈(흉골), '
 '갈비뼈(늑골)를 말하며 이를 모두 동일한 부위로 본다. 2) “골반뼈의 뚜렷한 기형”이라 함은 아래의 경우 중'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 213},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000759',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
