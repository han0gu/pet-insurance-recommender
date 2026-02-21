from langchain_core.documents import Document

chunk = Document(
    page_content=('에 제3항에서 정한 보장개시일(책임개시일) 이후에 보험증권에 기재된 반려견이 보험\n'
 '기간 중에 사망한 경우 보험증권에 기재된 보험가입금액을 사망위로금으로 보험수익\n'
 '자에게 보상하여 드립니다.\n'
 '② 제1항의 사망은 동물병원에서 적법하게 시행된 안락사를 포함합니다. 단, 이 경우 동\n'
 '물병원에서 발급한 소견서를 제출하여야 합니다.\n'
 '③ 제1항의 손해에 대한 보장개시일(책임개시일)은 이 특별약관의 보험계약일(이하 「보\n'
 '험계약일」이라 합니다)부터 그 날을 포함하여 30일이 지난 날의 다음날로 합니다. 이'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
