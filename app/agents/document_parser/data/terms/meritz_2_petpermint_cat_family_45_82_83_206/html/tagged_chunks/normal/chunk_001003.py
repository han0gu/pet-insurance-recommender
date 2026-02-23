from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해판정기준</h1><br><p id='16' data-category='list' style='font-size:16px'>1) "
 '“체간골”이라 함은 어깨뼈(견갑골), 골반뼈(장골,<br>제2천추 이하의 천골, 미골, 좌골 포함), 빗장뼈(쇄<br>골), '
 '가슴뼈(흉골), 갈비뼈(늑골)를 말하며 이를 모두<br>동일한 부위로 본다.<br>2) “골반뼈의 뚜렷한 기형”이라 함은 아래의 경우 '
 "중</p><footer id='17' style='font-size:14px'>188</footer><h1 id='18'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001003',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
