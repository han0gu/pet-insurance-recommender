from langchain_core.documents import Document

chunk = Document(
    page_content=('합계액- ② 이 계약이 의무보험이 아니고 다른 의무보험이 있는 경우에는 다른 의무보험에서 보\n'
 '- 상되는 금액(피보험자가 가입을 하지 않은 경우에는 보상될 것으로 추정되는 금액)을\n'
 '- 차감한 금액을 손해액으로 간주하여 제1항에 의한 보상할 금액을 결정합니다.\n'
 '- ③ 피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 의한\n'
 '- 지급보험금 결정에는 영향을 미치지 않습니다.\n'
 '# 제11조 (대위권)① 회사가 보험금을 지급한 때(현물보상한 경우를 포함합니다)에는 회사는 지급한 보험금'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
