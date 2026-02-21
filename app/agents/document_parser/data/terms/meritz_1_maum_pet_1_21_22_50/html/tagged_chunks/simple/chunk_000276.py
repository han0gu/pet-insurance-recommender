from langchain_core.documents import Document

chunk = Document(
    page_content=('이유로 계약을 해지하<br>거나 보험금 지급을 거절하지 않습니다.<br>⑦ 보통약관 제28조(보험료의 납입을 연체하여 해지된 계약의 '
 '부활(효력회복))에 따라 이<br>계약이 부활이 이루어진 경우에는 부활계약을 제2항의 최초계약으로 봅니다.(부활(효력<br>회복)이 '
 "여러차례 발생된 경우에는 각각의 부활(효력회복)계약을 최초계약으로 봅니다)</p><h1 id='105' "
 "style='font-size:14px'>제17조(계약의 무효)</h1><br><p id='106' "
 "data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000276',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
