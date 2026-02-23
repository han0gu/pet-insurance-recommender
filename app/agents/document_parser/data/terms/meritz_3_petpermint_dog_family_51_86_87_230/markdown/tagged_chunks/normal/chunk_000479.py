from langchain_core.documents import Document

chunk = Document(
    page_content=('# \uf000 보상 관련 용어| 용어 | 정의 |\n'
 '| --- | --- |\n'
 '| 사고 | 사고라 함은 급격하게 발생하는 것을 포함하여 위험이 서서히, 계속적, 반복적 또는 누적적으 로 노출되어 그 결과로 발생한 '
 '신체장해나 재 물손해를 말합니다. |\n'
 '| 1회의 사고 | 1회의 사고라 함은 하나의 원인 또는 사실상 같은 종류의 위험에 계속적, 반복적 또는 누적 적으로 노출되어 그 결과로 '
 '발생한 사고로서 피보험자나 피해자의 수 또는 손해배상청구의 수에 관계없이 1회의 사고로 봅니다. |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000479',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
