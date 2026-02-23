from langchain_core.documents import Document

chunk = Document(
    page_content=('는 자 중에서 보험금의 청구대리인(2인 이내에서 지정하되, 2인 지정시 대표대리인을\n'
 '지정)(이하 “지정대리청구인”이라 합니다)으로 지정할 수 있습니다. 또한, 지정대리청\n'
 '구인은 제4조(지정대리청구인의 변경지정)에 의한 변경 지정 또는 보험금 청구시에도\n'
 '다음 각 호의 어느 하나에 해당하여야 합니다.- 1. 피보험자의 가족관계등록상의 배우자\n'
 '- 2. 피보험자의 3촌 이내의 친족\n'
 '② 제1항에도 불구하고 지정대리청구인이 지정된 이후에 제1조(적용대상)의 보험수익자'),
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
