from langchain_core.documents import Document

chunk = Document(
    page_content=('지급사유가 발생하는 때에 회사에 보험 금을 청구하여 받을 수 있는 사람을 '
 '말합니다.</td></tr><tr><td>보험증권</td><td>계약의 성립과 그 내용을 증명하기 위하여 회 사가 계약자에게 드리는 '
 '증서를 말합니다.</td></tr><tr><td>진단계약</td><td>계약을 체결하기 위하여 반려동물이 건강진단 을 받아야 하는 계약을 '
 '말합니다.</td></tr><tr><td>피보험자</td><td>반려동물의 소유와 관련하여 보험사고로 손해 를 입은 사람(법인인 경우에는 '
 '그 이사 또는 법인의 업무를 집행하는 그 밖의 기관)을'),
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
 'indexing': {'chunk_id': 'chunk_000269',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
