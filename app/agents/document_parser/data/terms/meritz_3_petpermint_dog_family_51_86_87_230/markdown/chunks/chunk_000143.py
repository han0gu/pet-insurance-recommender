from langchain_core.documents import Document

chunk = Document(
    page_content=('증권에 기재된 반려동물의 질병 또는 상해로 인한 손해를\n'
 '보장하기 위하여 체결됩니다.# 제2조(용어의 정의)이 특별약관에서 사용되는 용어의 정의는, 이 특별약관의\n'
 '다른 조항에서 달리 정의되지 않는 한 다음과 같습니다.# \uf000 계약관련 용어| 용어 | 정의 |\n'
 '| --- | --- |\n'
 '| 계약자 | 회사와 계약을 체결하고 보험료를 납입할 의무 를 지는 사람을 말합니다. |\n'
 '| 보험 수익자 | 보험금 지급사유가 발생하는 때에 회사에 보험 금을 청구하여 받을 수 있는 사람을 말합니다. |'),
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
