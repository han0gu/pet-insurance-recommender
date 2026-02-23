from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- |\n'
 '| 【예시】 두 보험회사와 계약을 체결하고 100만원의 손해가 발생한 경우 보험금 계산 예시는 아래와 같습 니다. A사만 가입한 경우 '
 'A사의 보상책임액이 90만원이고 B사만 가입한 경우 B사의 보상책임액이 60만원 인 경우, A사 : 100만원 × 90만원 / (90만원 '
 '+ 60만원) = 60만원 지급 B사 : 100만원 × 60만원 / (90만원 + 60만원) = 40만원 지급 |\n'
 '피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 의한 지급보험금'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
