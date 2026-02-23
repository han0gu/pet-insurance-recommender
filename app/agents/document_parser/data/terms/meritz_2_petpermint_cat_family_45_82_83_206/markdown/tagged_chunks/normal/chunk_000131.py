from langchain_core.documents import Document

chunk = Document(
    page_content=('여야 한다.79\uf000 회사는 약관의 뜻이 명백하지 않은 경우에는 계약자에게\n'
 '유리하게 해석합니다.\n'
 '\uf000 회사는 보험금을 지급하지 않는 사유 등 계약자나 피보\n'
 '험자에게 불리하거나 부담을 주는 내용은 확대하여 해석하\n'
 '지 않습니다.제43조(설명서 교부 및 보험안내자료 등의 효력)\uf000 회사는 일반금융소비자에게 청약을 권유하거나 일반금융\n'
 '소비자가 설명을 요청하는 경우 보험상품에 관한 중요한 사\n'
 '항을 계약자가 이해할 수 있도록 설명하고 계약자가 이해하\n'
 '였음을 서명(「전자서명법」 제2조 제2호에 따른 전자서명'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000131',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
