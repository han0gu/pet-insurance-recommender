from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사의 고의<br>또는 과실로 계약이 무효로 된 경우와 회사가 승낙 전에 무<br>효임을 알았거나 알 수 있었음에도 보험료를 '
 '반환하지 않은<br>경우에는 보험료를 납입한 날의 다음날부터 반환일까지의<br>기간에 대하여 회사는 보험계약대출이율을 연단위 '
 "복리로<br>계산한 금액을 더하여 돌려 드립니다.</p><br><p id='3' data-category='list' "
 "style='font-size:16px'>① 타인의 사망을 보험금 지급사유로 하는 계약에서 계약<br>을 체결할 때까지 피보험자의 "
 '서면(「전자서명법」'),
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
 'indexing': {'chunk_id': 'chunk_000148',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
