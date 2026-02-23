from langchain_core.documents import Document

chunk = Document(
    page_content=('colspan="2">용 어 풀 이 통신판매계약</td><td>관</td></tr></tbody></table><br><p '
 "id='217' data-category='paragraph' style='font-size:16px'>전화·우편·인터넷 등 통신수단을 "
 "이용하여 체결하는 계약을 말합니다.</p><p id='218' data-category='paragraph' "
 "style='font-size:16px'>용 어 풀 이 자필서명<br>계약자가 성명기입란에 본인의 성명을 기재하고, 날인란에 "
 '사인(signature) 또 특별<br>는'),
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
