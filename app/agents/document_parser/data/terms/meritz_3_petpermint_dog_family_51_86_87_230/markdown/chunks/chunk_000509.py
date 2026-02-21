from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항 제1호에도 불구하고 다음 중 한가지의 경우에 해\n'
 '당되는 때에는 회사는 계약을 해지할 수 없습니다.- ① 회사가 최초계약 체결당시에 그 사실을 알았거나 과실\n'
 '- 로 알지 못하였을 때\n'
 '- ② 회사가 그 사실을 안 날부터 1개월 이상 지났거나 또\n'
 '- 는 제1회 보험료를 받은 때부터 보험금 지급사유가 발\n'
 '- 생하지 않고 2년이 지났을 때\n'
 '- ③ 최초계약을 체결한 날부터 3년이 지났을 때\n'
 '- ④ 보험설계사가 계약자 또는 피보험자에게 알릴 기회를\n'
 '- 주지 않았거나 계약자 또는 피보험자가 사실대로 알리'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
