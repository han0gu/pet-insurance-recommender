from langchain_core.documents import Document

chunk = Document(
    page_content=("id='77' data-category='paragraph' style='font-size:16px'>보험계약에 관한 전문성, 자산규모 "
 '등에 비추어 보험계<br>약에 따른 위험감수능력이 있는 자로서, 국가, 지방자치<br>단체, 한국은행, 금융회사, 주권상장법인 등을 '
 '포함하며<br>「금융소비자보호에 관한 법률」제2조(정의) 제9호에서<br>정하는 전문금융소비자를 말합니다.</p><br><p '
 "id='78' data-category='paragraph' style='font-size:16px'>\uf000 제1항에도 불구하고 "
 '청약한 날부터'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
