from langchain_core.documents import Document

chunk = Document(
    page_content=('그 사실을 알았거나 중대한 과실로 인하여 알지 못한 때\n'
 '에는 그러하지 아니하다.# 【상법 제651조의2(서면에 의한 질문의 효력)】보험자가 서면으로 질문한 사항은 중요한 사항으로 추정\n'
 '한다.# 제8조(계약 후 알릴 의무)\uf000 계약자 또는 피보험자는 보험기간 중에 다음 각 호의 변\n'
 '경이 발생한 경우에는 우편, 전화, 방문 등의 방법으로 지\n'
 '체없이 회사에 알려야 합니다.- ① 청약서의 기재사항을 변경하고자 할 때 또는 변경이\n'
 '- 생겼음을 알았을 때\n'
 '- ② 이 계약에서 보장하는 위험과 동일한 위험을 보장하는'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000169',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
