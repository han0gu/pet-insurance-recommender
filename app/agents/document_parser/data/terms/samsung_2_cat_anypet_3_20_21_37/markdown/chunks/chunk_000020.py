from langchain_core.documents import Document

chunk = Document(
    page_content=('【고양이백혈병 바이러스감염증】 고양이 백혈병바이러스에 감염에 의한 조혈기 질환\n'
 '【잔존유치】 영구치가 났는데도 불구하고 유치가 남아있어서 발치를 하는 경우\n'
 '【잠복고환】 고환이 음낭까지 내려오지 못하는 증상# 제6조(손해의 통지 및 조사)- ① 계약자 또는 피보험자는 제4조(보상하는 손해)에서 '
 '정한 사고가 생긴 것을 안 때에는 지체없이 그\n'
 '- 사실을 회사에 알려야 합니다.\n'
 '- ② 계약자 또는 피보험자가 제1항의 통지를 게을리하여 손해가 증가된 때에는 회사는 그 증가된 손해\n'
 '- 는 보상하여 드리지 않습니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
