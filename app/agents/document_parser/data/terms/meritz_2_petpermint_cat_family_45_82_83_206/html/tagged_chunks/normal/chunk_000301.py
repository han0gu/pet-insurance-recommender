from langchain_core.documents import Document

chunk = Document(
    page_content=('신청은 이 약관의「분쟁의 조정」조항에 따르<br>며 분쟁조정 신청 대상기관은 금융감독원의 금융분쟁조<br>정위원회를 '
 "말합니다.</p><br><p id='30' data-category='paragraph' "
 "style='font-size:20px'>\uf000 제2항에 따라 추가적인 조사가 이루어지는 경우, 회사는<br>피보험자의 청구에 따라 "
 "회사가 추정하는 보험금의 50% 상<br>당액을 가지급보험금으로 지급합니다.</p><br><h1 id='31' "
 "style='font-size:20px'>【가지급보험금】</h1><br><p id='32'"),
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
 'indexing': {'chunk_id': 'chunk_000301',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
