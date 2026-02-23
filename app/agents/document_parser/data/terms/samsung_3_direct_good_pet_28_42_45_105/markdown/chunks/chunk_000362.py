from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 납입최고(독촉)기간이 끝나는 날까지 보험료를 납입하지 않을 경우 납입최고(독촉)\n'
 '- 기 간이 끝나는 날의 다음날에 특별약관이 해지된다는 내용\n'
 '- 3. 계약자가 회사로부터 보험계약대출을 받은 경우 특별약관이 해지되는 즉시 해약환\n'
 '- 급금에서 보험계약대출원금과 이자가 차감된다는 내용\n'
 '- ② 납입최고(독촉)기간의 마지막 날이 영업일이 아닌 때에는 최고(독촉)기간은 그 다음\n'
 '- 날 까지로 합니다.\n'
 '- ③ 타인을 위한 계약의 경우 특정된 타인에게도 제1항 및 제2항에 따른 내용을 알려 드\n'
 '- 립니다.'),
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
