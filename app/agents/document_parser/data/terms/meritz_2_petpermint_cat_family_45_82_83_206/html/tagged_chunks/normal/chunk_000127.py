from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약자는 서면 등을 발송한<br>때에 그 발송 사실을 회사에 지체없이 알려야 합니다.<br>\uf000 계약자가 청약을 철회한 때에는 '
 "회사는 청약의 철회를<br>접수한 날부터 3영업일 이내에 납입한 보험료를 돌려드리</p><footer id='79' "
 "style='font-size:14px'>64</footer><p id='80' data-category='paragraph' "
 "style='font-size:16px'>며, 보험료 반환이 늦어진 기간에 대하여는 이 계약의 보험<br>계약대출이율을 연단위 복리로 "
 '계산한 금액을 더하여'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000127',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
