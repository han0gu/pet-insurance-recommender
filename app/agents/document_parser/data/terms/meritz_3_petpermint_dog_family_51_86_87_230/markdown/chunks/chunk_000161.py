from langchain_core.documents import Document

chunk = Document(
    page_content=('이내에 보험금을 지급합니다.\n'
 '\uf000 회사가 보험금 지급사유를 조사ㆍ확인하기 위해 필요한\n'
 '기간이 제1항의 지급기일을 초과할 것이 명백히 예상되는\n'
 '경우에는 그 구체적인 사유와 지급예정일 및 보험금 가지급\n'
 '제도(회사가 추정하는 보험금의 50% 이내를 지급)에 대하여\n'
 '피보험자에게 즉시 통지합니다. 다만, 지급예정일은 다음\n'
 '각 호의 어느 하나에 해당하는 경우를 제외하고는 제4조(보\n'
 '험금의 청구)에서 정한 서류를 접수한 날부터 30영업일 이\n'
 '내에서 정합니다.- ① 소송제기\n'
 '- ② 분쟁조정 신청\n'
 '- ③ 수사기관의 조사'),
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
