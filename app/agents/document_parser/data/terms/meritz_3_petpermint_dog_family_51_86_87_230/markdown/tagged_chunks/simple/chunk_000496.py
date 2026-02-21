from langchain_core.documents import Document

chunk = Document(
    page_content=('- 권리의 보전 또는 행사를 위한 필요한 조치를 취하는\n'
 '- 일\n'
 '- ③ 손해배상책임의 전부 또는 일부에 관하여 지급(변제),\n'
 '- 승인 또는 화해를 하거나 소송, 중재 또는 조정을 제\n'
 '- 기하거나 신청하고자 할 경우에는 미리 회사의 동의를\n'
 '- 받는 일\n'
 '\uf000 계약자 또는 피보험자가 정당한 이유없이 제1항의 의무\n'
 '를 이행하지 않았을 때에는 그 손해에서 다음의 금액을 뺍\n'
 '니다.- ① 제1항 제1호의 경우에는 그 노력을 하였더라면 손해를\n'
 '- 방지 또는 경감할 수 있었던 금액\n'
 '- ② 제1항 제2호의 경우에는 제3자로부터 손해의 배상을'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000496',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
