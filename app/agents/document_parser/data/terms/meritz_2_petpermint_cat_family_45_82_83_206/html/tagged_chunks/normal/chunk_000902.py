from langchain_core.documents import Document

chunk = Document(
    page_content=('. 가산이율 적용시 제8조(보험금의 지급절차) 제2<br>항 각 호의 어느 하나에 해당되는 사유로 지연<br>된 경우에는 해당기간에 '
 "대하여 가산이율을 적</p><footer id='28' style='font-size:14px'>174</footer><h1 "
 "id='29' style='font-size:20px'>용하지 않습니다.</h1><br><p id='30' "
 "data-category='list' style='font-size:16px'>5"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000902',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
