from langchain_core.documents import Document

chunk = Document(
    page_content=('보험자가 서면으로 질문한 사항은 중요한 사항으로 추정 한다.\n'
 '【사례】\n'
 '계약 청약을 하면서 보험설계사에게 고혈압이 있다고만 얘기하였을 뿐, 청약서의 계약 전 알릴 사항에 아무런 기재도 하지 않았을 경우에는 '
 '보험설계사에게만 고혈압 병력을 얘기하였다고 하더라도 회사는 계약 전 알릴 의 무 위반을 이유로 계약을 해지하고 보험금을 지급하지 않을 수 '
 '있습니다.\n'
 '제16조(상해보험계약 후 알릴 의무)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 62},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000053',
              'chunk_char_len': 217,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
