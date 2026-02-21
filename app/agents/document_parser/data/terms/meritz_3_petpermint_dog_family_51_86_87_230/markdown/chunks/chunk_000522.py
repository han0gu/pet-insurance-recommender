from langchain_core.documents import Document

chunk = Document(
    page_content=('권에 기재된 반려견의 행위에 기인하는 우연한 사고로 인하\n'
 '여 피해자에게 신체의 장해에 대한 법률상의 배상책임 또는\n'
 '타인 소유의 반려동물에 손해를 입혀 그에 대한 법률상의\n'
 '배상책임을 부담함으로써 입은 손해(이하「배상책임손해」\n'
 '라 합니다)를 보상합니다.\uf000 제1항의 피보험자라 함은 아래에 정한 보험증권에 기재# 된 피보험자 및 그 가족을 말합니다.- ① '
 '보험증권에 기재된 피보험자(이하「피보험자 본인」이\n'
 '- 라 합니다)\n'
 '- ② 피보험자 본인의 가족관계등록상 또는 주민등록상에\n'
 '- 기재된 배우자(이하「배우자」라 합니다)'),
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
