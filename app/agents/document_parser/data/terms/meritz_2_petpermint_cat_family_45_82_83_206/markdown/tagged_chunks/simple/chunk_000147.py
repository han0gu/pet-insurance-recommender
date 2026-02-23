from langchain_core.documents import Document

chunk = Document(
    page_content=('| 질병 | 상해를 제외한 상병을 모두 포함합니다. |\n'
 '| 중요한 사항 | 계약 전 알릴 의무와 관련하여 회사가 그 사실 을 알았더라면 계약의 청약을 거절하거나 보험 가입금액 한도 제한, 일부 '
 '보장 제외, 보험금 삭감, 보험료 할증과 같이 조건부로 승낙하는 등 계약 승낙에 영향을 미칠 수 있는 사항을 말합니다. |\n'
 '# \uf000 보상 관련 용어| 용어 | 정의 |\n'
 '| --- | --- |\n'
 '| 보험가입 금액 | 회사와 계약자간에 약정한 금액으로 보험사고 가 발생할 때 회사가 지급할 최대 보험금을 말 합니다. |'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000147',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
