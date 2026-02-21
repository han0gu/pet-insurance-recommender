from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제1항 내지 제2항에 해당하지 않는 시술(체외 충격파 쇄석술 및 변연절제를<br>동반하지 않은 단순 창상봉합술 등)<br>\uf000 '
 "제1항에서 의료기관이라 함은 의료법 제3조(의료기관) 제2항에서 정한 국내의 병</p><p id='115' "
 "data-category='paragraph' style='font-size:14px'>78 KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01)</p><br><table id='116' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>원이나 의원 또는"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
