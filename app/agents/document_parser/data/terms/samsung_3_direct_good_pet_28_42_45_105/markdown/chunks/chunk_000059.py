from langchain_core.documents import Document

chunk = Document(
    page_content=('회사가 건강상태 진단을 지원하는 계약, 보험기간이 90일 이내인 계약 또는 전문금융\n'
 'a , " 지기 해군 JUOL BIOL 二 ▲ ) 人 · ···- 34 -# [전문금융소비자]보험계약에 관한 전문성, 자산규모 등에 '
 '비추어 보험계약에 따른 위험감수능력이 있는 자로서,\n'
 '국가, 지방자치단체, 한국은행, 금융회사, 주권상장법인 등을 포함하며 「금융소비자 보호에 관한'),
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
