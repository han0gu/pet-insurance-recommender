from langchain_core.documents import Document

chunk = Document(
    page_content=('174| 배상책임 | 보험증권상의 보장지역 내에서 보험기간 중에 발생된 보험사고로 인하여 타인에게 입힌 손해 에 대한 법률상의 책임을 '
 '말합니다. |\n'
 '| --- | --- |\n'
 '| 법률상의 배상책임 | 법률상의 배상책임이라 함은 법률규정에 따른 배상책임을 말하며 계약에 의하여 법률규정보 다 가중된 '
 '배상책임(계약상의 가중책임)은 제 외합니다. |\n'
 '| 보상 한도액 | 회사와 계약자간에 약정한 금액으로 피보험자 가 법률상의 배상책임을 부담함으로써 입은 손 해 중 회사가 책임지는 금액의 '
 '최대 한도를 말 합니다. |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000480',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
