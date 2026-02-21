from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2) “영구적”이라 함은 원칙적으로 치유하는 때 장래 회\n'
 '- 복할 가망이 없는 상태로서 정신적 또는 육체적 훼손\n'
 '- 상태임이 의학적으로 인정되는 경우를 말한다.\n'
 '- 3) “치유된 후”라 함은 상해 또는 질병에 대한 치료의\n'
 '- 효과를 기대할 수 없게 되고 또한 그 증상이 고정된\n'
 '- 상태를 말한다.\n'
 '- 4) 다만, 영구히 고정된 증상은 아니지만 치료종결후 한시\n'
 '- 적으로 나타나는 장해에 대하여는 그 기간이 5년 이상\n'
 '- 인 경우 해당장해 지급률의 20%를 장해지급률로 한다.'),
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
