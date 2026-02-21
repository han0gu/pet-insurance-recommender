from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사가 나이의<br>착오를 발견하였을 때 이미 계약나이에 도달한 경우에<br>는 유효한 계약으로 보나, 제2호의 만15세 '
 "미만자에<br>관한 예외가 인정되는 것은 아닙니다.</p><br><h1 id='4' style='font-size:20px'>【상법 "
 "제731조(타인의 생명의 보험)】</h1><br><p id='5' data-category='paragraph' "
 "style='font-size:16px'>① 타인의 사망을 보험사고로 하는 보험계약에는 보험계<br>약 체결시에 그 타인의"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000152',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
