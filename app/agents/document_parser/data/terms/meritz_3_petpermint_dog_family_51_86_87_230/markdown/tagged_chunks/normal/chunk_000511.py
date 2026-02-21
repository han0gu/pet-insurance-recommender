from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자에게 지급합니다.\n'
 '\uf000 제1항 제1호에 따른 계약의 해지가 손해발생 후에 이루\n'
 '어진 경우에 회사는 그 손해를 보상하지 않으며, 계약 전\n'
 '알릴 의무 위반 사실(계약해지 등의 원인이 되는 위반사실\n'
 '을 구체적으로 명시)뿐만 아니라 계약 전 알릴 의무사항이\n'
 '중요한 사항에 해당되는 사유를「반대증거가 있는 경우 이\n'
 '의를 제기할 수 있습니다」라는 문구와 함께 계약자에게 서\n'
 '면 또는 전자문서 등으로 알려드립니다. 회사가 전자문서로\n'
 '안내하고자 할 경우에는 계약자에게 서면 또는 「전자서명'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000511',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
