from langchain_core.documents import Document

chunk = Document(
    page_content=('# 제 29조 (보험료의 납입이 연체되는 경우 납입최고(독촉)와 특별약관의 해지)① 계약자가 제2회 이후의 보험료를 납입기일까지 납입하지 '
 '않아 보험료 납입이 연체중\n'
 '인 경우에 회사는 14일(보험기간이 1년 미만인 경우에는 7일) 이상의 기간을 납입최\n'
 '고(독촉)기간으로 정하여 계약자에게 다음 각 호의 내용을 서면(등기우편 등), 전화(음\n'
 '성녹음) 또는 전자문서 등으로 알려 드립니다.- 1. 납입최고(독촉)기간 내에 연체보험료를 납입하여야 한다는 내용'),
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
