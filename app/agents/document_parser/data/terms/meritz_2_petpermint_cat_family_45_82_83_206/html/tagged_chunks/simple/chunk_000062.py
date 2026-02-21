from langchain_core.documents import Document

chunk = Document(
    page_content=('알려드리며, 만기환급금을 지급함에 있어<br>지급일까지의 기간에 대한 이자의 계산은【별표1(보험금을<br>지급할 때의 적립이율 계산)】에 '
 '따릅니다.<br>\uf000 보험료 납입기간 중에 제2조(용어의 정의)에서 정한 적<br>립보험료를 감액하거나 중도인출을 하는 경우 '
 "제1항의 만기<br>환급금은 가입시점의 예상금액보다 감소할 수 있습니다.</p><h1 id='85' "
 "style='font-size:20px'>제11조(보험금 받는 방법의 변경)</h1><br><p id='86' "
 "data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000062',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
